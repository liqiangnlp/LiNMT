#!/usr/bin/perl -w


# * NiuTrans.NMT v1.0                   * #
# * Author: Qiang Li,                   * #
# * Email : liqiangneu@gmail.com        * #
# * Date  : 10/30/2016                  * #
# * Time  : 15:31                       * #
# * Char seg following word seg         * #


use strict;
use Encode;

my $logo = "Character segmenter following word segmenter for Chinese\n";

print STDERR $logo;

my %param;


GetParameter(@ARGV);
CharSegmenter();


sub CharSegmenter
{
  open(SRCINFILE, "<:encoding(UTF-8)", $param{"-chn"}) or die "Error: can not read file $param{ '-chn' }.\n";
  open(SRCOUTFILE, ">:encoding(UTF-8)", $param{"-outchn"}) or die "Error: can not write file $param{'-outchn'}.\n";
  
  my $line_num = 0;
  while (my $line_src = <SRCINFILE>)
  {
    $line_src =~ s/[\r\n]//g;
    ++$line_num;
    
    my @chinese_words = split /\s+/, $line_src;
    my $final_src = "";
    foreach my $word (@chinese_words)
    {
      #if ($word =~ /^[[:ascii:]]+/)
      if ($word =~ /^[[:ascii:]]+$/)
      {
        $final_src .= $word." ";
        next;
      }
      
      my @chinese_chars = split //, $word;
      foreach my $char (@chinese_chars)
      {
        $final_src .= $char." ";
      }
    }

    $final_src =~ s/[\r\n]//g;
    $final_src =~ s/\s+$//g;
    $final_src =~ s/^\s+//g;
    
    print SRCOUTFILE $final_src."\n";
    
    if ($line_num % 10000 == 0)
    {
      print STDERR "\r  $line_num";
    }
  }
  print STDERR "\r  $line_num\n";

  close(SRCINFILE);
  close(SRCOUTFILE);
}


sub GetParameter
{
  if (( scalar( @_ ) < 4 ) || ( scalar( @_ ) % 2 != 0 ))
  {
    print STDERR "[USAGE]\n".
                 "         NiuTrans.NMT-chn-cs-follow-ws.pl             [OPTIONS]\n".
                 "[OPTION]\n".
                 "          -chn    :  Input chinese file.\n".
                 "          -outchn :  Output filtered chinese file.\n".
                 "[EXAMPLE]\n".
                 "      perl NiuTrans.NMT-chn-cs-follow-ws.pl      [-chn    FILE]\n".
                 "                                                 [-outchn FILE]\n";
    exit(0);
  }
          
  my $pos;
  for( $pos = 0; $pos < scalar( @_ ); ++$pos )
  {
    my $key = $ARGV[ $pos ];
    ++$pos;
    my $value = $ARGV[ $pos ];
    $param{ $key } = $value;
  }
          
  if( !exists $param{ "-chn" } )
  {
    print STDERR "Error: please assign \"-chn\"!\n";
    exit( 1 );
  }

  if( !exists $param{ "-outchn" } )
  {
    print STDERR "Error: please assign \"-outchn\"!\n";
    exit( 1 );
  }

}
